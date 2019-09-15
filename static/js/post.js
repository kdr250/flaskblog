$(function() {

  function buildHTML(post){
    var html = `<div id="posts">
    <h1>${post.title}</h1>
    <p>投稿日: ${post.date_posted} By ${post.authorname}</p>
    <p>${post.content}</p>
    </div>`
    return html
  }

  $('#submit').on('click', function(e){
    e.preventDefault();
    console.log('hoge');
    $.ajax({
      type: 'POST',
      url: '/post_ajax',
      data: $(this).parent().parent('form').serialize(),
    })
    .done(function (data) {
      console.log('Ajax成功!')
      var html = buildHTML(data)
      $('#message-box').append(html)
      $('#post-form').get(0).reset();
      $('html, body').animate({scrollTop:$('#message-box')[0].scrollHeight});
    })
    .fail(function() {
      alert('Ajaxに失敗しました');
    })
  })
})